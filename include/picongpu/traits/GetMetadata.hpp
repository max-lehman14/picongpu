/* Copyright 2024 Julian Lenz
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <type_traits>

#include <nlohmann/json.hpp>


namespace picongpu
{
    namespace traits
    {
        template<typename, typename = void>
        inline constexpr bool hasMetadata = false;

        template<typename T>
        inline constexpr bool hasMetadata<T, std::void_t<decltype(std::declval<T>().metadata())>> = true;

        template<typename, typename = void>
        inline constexpr bool hasMetadataCT = false;

        template<typename T>
        inline constexpr bool hasMetadataCT<T, std::void_t<decltype(T::metadata())>> = true;

        template<typename, typename = void>
        inline constexpr bool hasMetadataRT = false;

        template<typename T>
        inline constexpr bool hasMetadataRT<T, std::enable_if_t<hasMetadata<T> && !hasMetadataCT<T>>> = true;

        // doc-include-start: GetMetdata trait
        template<typename TObject, typename = void>
        struct GetMetadata
        {
            static_assert(
                false,
                "If you reached this point, you tried to register a type or object for metadata output that does not "
                "supply the necessary information. Try to specialise picongpu::traits::GetMetadata for your type or "
                "add a `.metadata()` method to it.");
        };

        template<typename TObject>
        struct GetMetadata<TObject, std::enable_if_t<hasMetadataRT<TObject>>>
        {
            // holds a constant reference to the RT instance it's supposed to report about
            TObject const& obj;

            nlohmann::json description() const
            {
                return obj.metadata();
            }
        };

        template<typename TObject>
        struct GetMetadata<TObject, std::enable_if_t<hasMetadataCT<TObject>>>
        {
            nlohmann::json description() const
            {
                return TObject::metadata();
            }
        };
        // doc-include-end: GetMetdata trait
    } // namespace traits
} // namespace picongpu
